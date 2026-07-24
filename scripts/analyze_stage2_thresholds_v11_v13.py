import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path
from sklearn.model_selection import GroupKFold
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.models.evaluate import find_best_thresholds_multiclass, evaluate_multiclass_at_thresholds

npz = np.load('data/features/stage2_multiclass_features.npz')
X, y, groups = npz['X'].astype(np.float32), npz['y'].astype(np.int64), npz['groups']

print('Dataset shape', X.shape, 'classes', np.unique(y, return_counts=True))

for exp in ['v11', 'v12', 'v13']:
    exp_dir = Path(f'experiments/stage2_mlp_features_v2.3_focal_smote_{exp}')
    if not exp_dir.exists():
        continue
    print(f'\n=== {exp} ===')
    gkf = GroupKFold(n_splits=5)
    fold_idx = 0
    f1_f_scores = []
    f1_macro_scores = []
    f1_s_scores = []
    f1_v_scores = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        fold_dir = exp_dir / f'fold_{fold_idx}'
        model_path = fold_dir / 'model.keras'
        scaler_path = fold_dir / 'input_scaler.pkl'
        if not model_path.exists() or not scaler_path.exists():
            fold_idx += 1
            continue
        model = tf.keras.models.load_model(str(model_path), compile=False)
        scaler = joblib.load(scaler_path)
        X_val = scaler.transform(X[val_idx])
        y_val = y[val_idx]
        proba = model.predict(X_val, batch_size=4096, verbose=0)
        best = find_best_thresholds_multiclass(y_val, proba, class_names=['S','V','F'], metric='F1_macro', search_step=0.01, fallback_class=1)
        f1_f = best['per_class']['F']['F1']
        f1_macro = best['global']['F1_macro']
        f1_s = best['per_class']['S']['F1']
        f1_v = best['per_class']['V']['F1']
        f1_f_scores.append(f1_f)
        f1_macro_scores.append(f1_macro)
        f1_s_scores.append(f1_s)
        f1_v_scores.append(f1_v)
        print(f'  fold {fold_idx}: thresholds={best["thresholds"]} F1-macro={f1_macro:.4f} F1(S)={f1_s:.4f} F1(V)={f1_v:.4f} F1(F)={f1_f:.4f}')
        fold_idx += 1
    if f1_f_scores:
        print(f'  mean: F1-macro={np.mean(f1_macro_scores):.4f}±{np.std(f1_macro_scores):.4f} F1(S)={np.mean(f1_s_scores):.4f}±{np.std(f1_s_scores):.4f} F1(V)={np.mean(f1_v_scores):.4f}±{np.std(f1_v_scores):.4f} F1(F)={np.mean(f1_f_scores):.4f}±{np.std(f1_f_scores):.4f}')

# Also try optimizing explicitly for F1(F) on v11 best model? We can brute force threshold search for F1(F) on one representative fold.
print('\n=== Brute-force F1(F) optimization on v11 fold 3 (best F fold) ===')
exp_dir = Path('experiments/stage2_mlp_features_v2.3_focal_smote_v11')
fold_dir = exp_dir / 'fold_3'
model = tf.keras.models.load_model(str(fold_dir / 'model.keras'), compile=False)
scaler = joblib.load(fold_dir / 'input_scaler.pkl')
gkf = GroupKFold(n_splits=5)
for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    if fold_idx != 3:
        continue
    X_val = scaler.transform(X[val_idx])
    y_val = y[val_idx]
    proba = model.predict(X_val, batch_size=4096, verbose=0)
    best_f1_f = 0
    best_thr = None
    best_other = None
    for t_s in np.arange(0.1, 0.9, 0.05):
        for t_v in np.arange(0.1, 0.9, 0.05):
            for t_f in np.arange(0.05, 0.95, 0.05):
                res = evaluate_multiclass_at_thresholds(y_val, proba, thresholds={'S': t_s, 'V': t_v, 'F': t_f}, class_names=['S','V','F'], fallback_class=1, fallback_to_argmax=True)
                f1_f = res['per_class']['F']['F1']
                if f1_f > best_f1_f:
                    best_f1_f = f1_f
                    best_thr = {'S': t_s, 'V': t_v, 'F': t_f}
                    best_other = (res['global']['F1_macro'], res['per_class']['S']['F1'], res['per_class']['V']['F1'])
    print(f'Best F1(F)={best_f1_f:.4f} at thresholds={best_thr} F1-macro={best_other[0]:.4f} F1(S)={best_other[1]:.4f} F1(V)={best_other[2]:.4f}')
