import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('style_control',Path('scripts/build_sft_style_control.py'))
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def test_normalization_preserves_string_contents_and_ast():
    source='if True:\n\ts = """abc\n\tliteral tab\n"""\n\tprint(s)\n'
    result=mod.normalize(source)
    assert '\n    s =' in result and '\n    print(s)' in result
    assert '\n\tliteral tab\n' in result
