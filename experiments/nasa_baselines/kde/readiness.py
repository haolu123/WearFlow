"""Evidence gate only. This is not an implementation of Kim et al.'s model."""
import json
from pathlib import Path

class MissingPrimaryMethodError(RuntimeError):
    pass

def load_spec(path=None):
    return json.loads(Path(path or Path(__file__).with_name("mapping.json")).read_text(encoding="utf-8"))

def require_reproducible(spec=None):
    spec = load_spec() if spec is None else spec
    missing = spec.get("missing_core", [])
    if missing:
        raise MissingPrimaryMethodError(
            "Kim et al. KDE baseline cannot enter formal training: missing primary definitions: "
            + ", ".join(missing)
        )
    if not spec.get("equation_mapping_verified", False):
        raise MissingPrimaryMethodError("Paper-to-code equation mapping has not been verified.")
    return True

if __name__ == "__main__":
    require_reproducible()

