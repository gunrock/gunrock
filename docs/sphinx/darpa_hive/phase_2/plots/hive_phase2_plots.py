#!/usr/bin/env python3.8

# run from gunrock-docs/docs/hive_phase2/plots

import altair as alt
import pandas as pd
import numpy
import datetime
import sys

sys.path.insert(1, "../../../../gunrock-io/scripts")

from fileops import save, getChartHTML
from filters import *
from logic import *

alt.data_transformers.disable_max_rows()


def normalizeBy1GPU(df, dest, quantityToNormalize, columnsToGroup):
    # http://stackoverflow.com/questions/41517420/pandas-normalize-values-within-groups-with-one-reference-value-per-group-group#41517726
    df1 = df.loc[df["num-gpus"] == 1, columnsToGroup + [quantityToNormalize]]
    suffix = "_1"
    dfmerge = df.merge(df1, on=columnsToGroup, how="left", suffixes=["", suffix])
    dfmerge[dest] = dfmerge[quantityToNormalize + suffix] / dfmerge[quantityToNormalize]
    return dfmerge


def fileIsNotA100(f):
    return "dgxa100" not in f


# slurp up all JSONs
roots = ["../results/"]
# roots = ["../results/vn_eval_mgpu/dgxa100_40g_1tb/"]
df = filesToDF(roots=roots, fnFilterInputFiles=[fileEndsWithJSON, fileIsNotA100])
# df["num-gpus"] = df["jsonfile"].str.extract(r"GPU(\d+)")
# df["num-gpus"] = pd.to_numeric(df["num-gpus"])
df.rename({"walkmode": "variant"}, axis="columns", inplace=True)

# this next for loop unpacks the "tag" column into separate columns
# it's iterative, that's not ideal, but at least the iteration is in a nice loop comprehension
for col in ["num-gpus", "variant"]:
    df[col] = [
        [tag.split(":")[1] for tag in l if tag.split(":")[0] == col] for l in df["tag"]
    ]
    df[col] = df[col].explode()  # list to scalar
for col in ["num-gpus"]:  # numeric cols only
    df[col] = pd.to_numeric(df[col])

df = df.sort_values(by=["primitive", "dataset", "variant", "num-gpus"]).reset_index(
    drop=True
)
dfs = {}
tables = {}
charts = {}

tablespage = open("../tables/README.md", "w")
tablespage.write("# Tables of results\n\n")

# get list of primitives
prims = df["primitive"].unique()
for prim in prims:
    with open(f"{prim}_plots.md", "w") as plotlist:
        plotlist.write(f'## Scalability plots\n\n![{prim}: speedup over 1 GPU vs. number of GPUs](plots/{prim}.pdf "{prim}")\n\n')
    dfs[prim] = df[df["primitive"] == prim]
    dfs[prim] = dfs[prim].dropna(axis=1, how="all")  # drop columns if all n/a

    # join on columns in "cols"
    cols = ["primitive", "dataset"]
    has_variants = "variant" in dfs[prim].columns  # multiple variants
    if has_variants:
        cols.append("variant")
    # compute speedup for each group of "cols"
    dfs[prim] = normalizeBy1GPU(
        df=dfs[prim],
        dest="speedup",
        quantityToNormalize="avg-process-time",
        columnsToGroup=cols,
    )
    # now add back in columns we care about
    cols.extend(["num-gpus", "avg-process-time", "speedup"])
    tables[prim] = dfs[prim][cols]
    if has_variants:
        with open(f"{prim}_plots.md", "a") as plotlist:
            for variant in dfs[prim]["variant"].unique().tolist():
                name = prim + "_" + variant
                dfs[name] = dfs[prim][dfs[prim]["variant"] == variant]
                tables[name] = dfs[name][cols]
                tablefilename = f"../tables/{name}.md"
                tables[name].to_markdown(buf=tablefilename, index=False)
                with open(tablefilename, 'a') as tablefile:
                    tablefile.write(f"\n\n: Tabular data for {name}")
                tables[name].to_html(
                    buf=f"../tables/{name}.html", index=False, border=0
                )
                plotlist.write(f'![{name}: speedup over 1 GPU vs. number of GPUs](plots/{name}.pdf "{name}")\n\n')
                tablespage.write(f"[{name}](hive_phase2/tables/{name}.html)\n\n")
    else:
        # only write top-level table if there are no subtables
        tablefilename = f"../tables/{prim}.md"
        tables[prim].to_markdown(buf=tablefilename, index=False)
        with open(tablefilename, 'a') as tablefile:
            tablefile.write(f"\n\n: Tabular data for {prim}")
        tables[prim].to_html(buf=f"../tables/{prim}.html", index=False, border=0)
        tablespage.write(f"[{prim}](hive_phase2/tables/{prim}.html)\n\n")
tablespage.close()


# now start plotting all tables
for table in tables:
    charts[table] = (
        alt.Chart(tables[table], mark="point")
        .encode(
            x=alt.X(
                "num-gpus",
                type="ordinal",
                axis=alt.Axis(
                    title="Number of GPUs",
                ),
                scale=alt.Scale(type="linear"),
            ),
            y=alt.Y(
                "speedup",
                type="quantitative",
                axis=alt.Axis(
                    title=f"{table} Speedup over 1 GPU",
                ),
                scale=alt.Scale(type="linear"),
            ),
            shape=alt.Shape(
                "dataset",
                type="nominal",
            ),
        )
        .interactive()
    )
    has_variants = "variant" in tables[table].columns  # multiple variants
    if has_variants:
        charts[table] = charts[table].encode(
            color=alt.Color(
                "variant",
                type="nominal",
            ),
        )

    save(
        chart=charts[table],
        df=tables[table],
        plotname=table,
        outputdir="./",
        formats=["png", "pdf", "json"],
    )


with open(f"README.md", "w") as plotspage:
    plotspage.write(
        """## Scalability plots
<script type="text/javascript">
  var svgopt = { renderer: "svg" }
"""
    )
    divs = ""
    for chart in sorted(charts):
        chart_id = chart.replace("-", "_")

        # note to self: { } in fstrings need to be {{ }}
        plotspage.write(
            f"""
  var spec_{chart_id} = "https://raw.githubusercontent.com/gunrock/docs/master/docs/hive_phase2/plots/{chart}.json";
  vegaEmbed('#vis_{chart_id}', spec_{chart_id}, opt=svgopt).then(function(result) {{
    // Access the Vega view instance (https://vega.github.io/vega/docs/api/view/) as result.view
  }}).catch(console.error);

"""
        )
        divs += f"""
<div id="vis_{chart_id}"></div>

"""
    plotspage.write("</script>\n\n")
    plotspage.write(divs)
