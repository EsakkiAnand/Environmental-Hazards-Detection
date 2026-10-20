import ultralytics.engine.trainer as trainer_mod
import torch
import pathlib

print("trainer_mod attributes:")
print("has torch:", hasattr(trainer_mod, "torch"))
print("has torch_save:", hasattr(trainer_mod, "torch_save"))

if hasattr(trainer_mod, "torch_save"):
    print("trainer_mod.torch_save is:", trainer_mod.torch_save)
