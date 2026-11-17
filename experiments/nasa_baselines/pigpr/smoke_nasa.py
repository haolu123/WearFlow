"""One fixed-parameter inner-split check, never outer-test selection."""
from pathlib import Path
import hashlib,json,pickle,time
import numpy as np
from sun_pigpr import SunPIGPR
base=Path(__file__).resolve().parent
data=base.parent/"data/fold_a/inner_1"
z=np.load(data/"arrays.npz")
meta=json.loads((data/"metadata.json").read_text(encoding="utf-8"))
tr=np.isin(z["case"],meta["split"]["train"]) & z["sensor_valid"]
va=np.isin(z["case"],meta["split"]["validation"]) & z["sensor_valid"]
t0=time.perf_counter()
model=SunPIGPR().fit(z["sensor_z"][tr],z["y_mm"][tr],z["time"][tr],
                     z["case"][tr],z["X_ct"][tr,:4])
p=model.predict(z["sensor_z"][va],z["time"][va],z["case"][va],z["X_ct"][va,:4])
reload=pickle.loads(pickle.dumps(model))
p2=reload.predict(z["sensor_z"][va],z["time"][va],z["case"][va],z["X_ct"][va,:4])
np.testing.assert_array_equal(p,p2)
label=np.isfinite(z["y_mm"][va])
rows=[{"case":int(c),"run":int(run),"VB":float(y) if np.isfinite(y) else None,"prediction":float(pred)}
      for c,run,y,pred in zip(z["case"][va],z["run"][va],z["y_mm"][va],p)]
report={"status":"passed","experiment":"inner smoke only, no hyperparameter search",
        "training_cases":meta["split"]["train"],"validation_cases":meta["split"]["validation"],
        "outer_test_evaluated":False,"train_valid_records":int(tr.sum()),
        "validation_valid_records":int(va.sum()),"validation_labels":int(label.sum()),
        "validation_mae_mm":float(np.mean(np.abs(p[label]-z["y_mm"][va][label]))),
        "finite":bool(np.isfinite(p).all()),"reload_exact":True,
        "elapsed_seconds":time.perf_counter()-t0,
        "data_sha256":hashlib.sha256((data/"arrays.npz").read_bytes()).hexdigest(),
        "model":model.audit_state(),"predictions":rows}
(base/"nasa_smoke_results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k not in ["model","predictions"]},indent=2))

