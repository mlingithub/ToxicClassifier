import yaml,os

with open(".data/config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

for i in range(0,len(cfg)):
    if cfg[i]["name"]!="":
        if cfg[i]['source']['download']=="no" :
            os.system("cd .data \
                && git lfs install \
                && git config --global filter.lfs.process \"git-lfs filter-process --skip\" \
                && git submodule add {0}".format(cfg[i]['source']['url']))
        else:
            os.system("cd .data \
                && git config --global filter.lfs.process \"git-lfs filter-process --skip\" \
                && git submodule add {0} \
                && cd {1} \
                && git lfs pull".format(cfg[i]['source']['url'],cfg[i]['source']['url'].split("/")[-1].split(".git")[0]))
