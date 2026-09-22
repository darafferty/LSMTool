import lsmtool
from pathlib import Path
import pytest
import time
from python_callgraph_mermaid import callgraph
from python_callgraph_mermaid.mermaid import MermaidOutput
import cmasher as cmr
# import webbrowser


output_root = Path(__file__).parent
skymodels_path = output_root / "resources"
callgraph_path = output_root / "callgraphs"
callgraph_path.mkdir(exist_ok=True)


@pytest.mark.parametrize(
    "filename", [next(skymodels_path.rglob("sector_1.apparent_sky.txt"))]
)
@pytest.mark.parametrize("per_patch_projection", [True, False])
def test_get_patch_positions(filename, per_patch_projection):
    # applyBeam_group=False
    # filename = pytestconfig.resource_dir / "sector_1.apparent_sky.txt"
    source_skymodel = lsmtool.load(filename)
    n_sources = len(source_skymodel.table)
    print("Nsources", n_sources)

    filename = f"get_patch_positions-nsources_{n_sources}-per_patch_{per_patch_projection}.html"
    with callgraph(
        MermaidOutput(
            callgraph_path / filename,
            cmap="cmr.neon_r",
            # viewer=webbrowser,
            mermaid_config={"config": {"themeVariables": {"fontSize": "16px"}}},
        )
    ):
        start = time.time()
        positions = source_skymodel.getPatchPositions(
            perPatchProjection=per_patch_projection, method="wmean"
        )
        end = time.time()

    print("Elapsed time:", (dt := end - start))

    # print(positions)qnWY6KUW9DnDKx70BTu@
