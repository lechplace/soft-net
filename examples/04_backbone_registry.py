"""
Example 4: Explore BackboneRegistry — no fitting required.

Run:
    uv run python examples/04_backbone_registry.py
"""

from softnet.image import BackboneRegistry

print("=== All available backbones ===")
for name in BackboneRegistry.list():
    spec = BackboneRegistry.get_spec(name)
    print(f"  {name:<25} family={spec.family:<15} "
          f"default_input={spec.default_input_size}")

print("\n=== EfficientNet family only ===")
for name in BackboneRegistry.list(family="efficientnet"):
    print(f"  {name}")

print("\n=== Families ===")
print(BackboneRegistry.families())
