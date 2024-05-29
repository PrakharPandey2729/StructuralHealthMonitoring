import subprocess
import os
# Path to your ABAQUS executable
abaqus_path = r'C:\SIMULIA\Commands\abaqus.bat'  

# Path to your Python script
script_path = 'master.py'  

#damage_index = [1,9,11,13,14,7,2,32,33,43,44,45,56,58,21,20,15]
#damage_index = [181,189,187,185,184,183,188,182,186]
damage_index = [91,99,101,103,104,97,121,124,133,136,146,147,110,105]
#damage_index = [1,8,9]

for i in range(len(damage_index)):
    for j in range(i+1, len(damage_index)):
        # Set the environment variable
        if(i == j):
            continue
        os.environ['DAMAGE_INDEX'] = str([damage_index[i], damage_index[j]])
        #os.environ['DAMAGE_INDEX'] = [damage_index[i], damage_index[j]]
        subprocess.run([abaqus_path, 'cae', 'noGUI=' + script_path])
        #subprocess.run([abaqus_path, 'cae', 'script=' + script_path])

