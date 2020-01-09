import glob, os, sys
import nbformat as nbf
import datetime

repo_folder=sys.argv[1]
branch=sys.argv[2]

os.system ("git fetch origin && git reset --hard origin/master && git clean -f -d")

jupyterFiles=[]
from pathlib import Path
for filename in Path('./').rglob('*.ipynb'):
    jupyterFiles.append(filename)
for JupterFile in jupyterFiles:
    JupterFile=str(JupterFile)
    cmd="cd ~/../../ && data/anaconda/envs/py35/bin/jupyter-notebook list"
    p = os.popen(cmd).read()
    server="http"+(p.split(" ::"))[0].split("http")[1]
    server=server.split("?token")[0]+"tree/"+repo_folder+"/"+JupterFile+"?token"+server.split("?token")[1]
    f=open(JupterFile, "r")
    contents =f.read()

    if "cells" not in contents:
            import shutil
            p = os.popen("ls").read()
            shutil.copyfile("../config/template.ipynb", JupterFile)
            text = """<div style="text-align: right"><a href={0} target="_blank">Edit</a></div>""".format(server)
            message = "NB updated on " + str(datetime.datetime.now())
            data=nbf.read(JupterFile,4)
            data['cells']=[nbf.v4.new_markdown_cell(text)]+data['cells']
            with open(JupterFile, 'w') as f:
                nbf.write(data, f)
            os.system ("git add {0} && git commit -m \"{1}\" -m {2} && git push origin {3}".format(JupterFile,message,server,branch))

    data=nbf.read(JupterFile,4)
    if len(data["cells"])>0 :
        if "http://localhost" not in data["cells"][0]['source']:
            data=nbf.read(JupterFile,4)
            text = """<div style="text-align: right"><a href={0} target="_blank">Edit</a></div>""".format(server)
            data['cells']=[nbf.v4.new_markdown_cell(text)]+data['cells']
            message = "NB updated on " + str(datetime.datetime.now())
            with open(JupterFile, 'w') as f:
                nbf.write(data, f)
            os.system ("git add {0} && git commit -m \"{1}\" -m {2} && git push origin {3}".format(JupterFile,message,server,branch))
