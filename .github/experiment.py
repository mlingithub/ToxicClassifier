import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import markdown,os,sys

# get wiki folder path
home = str(Path.home())
wiki_folder=sys.argv[3]

#read md
os.system("cd {0} && git add . && git pull --all".format(home+'/Desktop/'+wiki_folder))
f = open(home+'/Desktop/'+wiki_folder+'/experiment.md', 'r')
htmlmarkdown=markdown.markdown( f.read() )
f.close()
existing_lines=pd.read_html(htmlmarkdown,index_col=False)[0]
existing_lines=existing_lines.iloc[:,1:]

#get data
dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
commit_string= sys.argv[1] #os.popen('git log -n 1 --pretty=format:"%H"').read()
branch_string = sys.argv[2]
results=pd.read_csv(".artifacts/result.csv",index_col=False)

#append data
aux=pd.DataFrame(np.array([[dt_string,commit_string,branch_string]]),columns=['time', 'commit', 'branch'])
new_line=pd.concat([aux,results],axis=1)
data=pd.concat((existing_lines,new_line),axis=0)
data=data.reset_index(drop=True) # Resets the index, makes factor a column

#write data
f = open(home+'/Desktop/'+wiki_folder+'/experiment.md', 'w')
f.write(data.to_html())
f.close()
os.system("cd {0} && git add . && git commit -m \"update\" && git push origin master".format(home+'/Desktop/'+wiki_folder))
