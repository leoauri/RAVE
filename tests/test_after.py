import pytest
from absl import app, flags
import sys, os

from . import FlagEnv

# check if environment variables are defined in .env
HAS_TEST_ENV_VARIABLES = True
if not os.environ.get("AFTER_DB_PATH"):
    HAS_TEST_ENV_VARIABLES = False
if not (os.environ.get("AFTER_CKPT_PATH") or os.environ.get("AFTER_TS_PATH")):
    HAS_TEST_ENV_VARIABLES = False

# populate models to test
test_models = []
if os.environ.get("AFTER_CKPT_PATH"):
    test_models.extend(os.environ.get("AFTER_CKPT_PATH").split(','))
if os.environ.get("AFTER_TS_PATH"):
    test_models.extend(os.environ.get("AFTER_TS_PATH").split(','))

# parse configurations
flag_configs = [
    FlagEnv(
        name = f"after_test_{FlagEnv.get_counter()}", 
    )
]


@pytest.mark.forked
@pytest.mark.parametrize("db_path", [os.environ.get("AFTER_DB_PATH")])
@pytest.mark.parametrize("model_path", test_models)
@pytest.mark.parametrize("flag_config", flag_configs)
@pytest.mark.skipif(not HAS_TEST_ENV_VARIABLES, reason="environment variables not found")
def test_after_scripting(model_path, db_path, flag_config):
    if not "main_after" in locals():
        from rave.scripts import train_after
        
    flag_config.db_path = db_path
    flag_config.model = model_path
    argv = []
    for v in flag_config: 
        argv.extend(v)
    train_after.parse_flags()
    app.run(train_after.main, argv=[sys.argv[0], *argv])
