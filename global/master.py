from abaqus import *
from abaqusConstants import *
backwardCompatibility.setValues(includeDeprecated=True,
                                reportDeprecated=False)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Path to your .cae file
cae_path = 'Bridge.cae'  # Replace with your path
# Open the .cae file
mdb = openMdb(pathName=cae_path)

myModel = mdb.models['Bridge']
myBridge = myModel.parts['All']
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import assembly
# Create a part instance.
myAssembly = myModel.rootAssembly
myInstance = myAssembly.instances['All-1']

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DEFINING STEP
import step
myModel.ImplicitDynamicsStep(name='Harmonic Force', previous='Gravity', initialInc = 0.01, maxInc = 0.01, minInc = 0.01, timePeriod=1.0)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DEFINING LOAD
import math
# Define a harmonic amplitude curve
time_values = [i * 0.01 for i in range(101)]  # replace with your actual time values
amplitude_values = [math.sin(2 * math.pi * t) for t in time_values]  # sine wave with frequency of 1 Hz
myModel.TabularAmplitude(name='HarmonicAmplitude', timeSpan = STEP, smooth= SOLVER_DEFAULT, data=zip(time_values, amplitude_values))

# Applying the harmonic amplitude via load 'Load-Harmonic'. Using all nodes in the instance. 
sregion = myAssembly.Set(nodes= myInstance.nodes, name='LoadSet')
myModel.ConcentratedForce(name='Load-Harmonic', createStepName='Harmonic Force', region=sregion, cf2=1.0, amplitude='HarmonicAmplitude')

myModel.FieldOutputRequest(name='Output-1', createStepName='Harmonic Force', 
                          variables=('U',))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DEFINING SECTION ASSIGNMENT
import os
import ast
damaged_elements = []  # Replace with your element labels
damage_index = []
damage_index_str = os.environ['DAMAGE_INDEX']
damage_index = ast.literal_eval(damage_index_str)
damaged_elements.append(myBridge.elements[damage_index[0]-1:damage_index[0]])
damaged_elements.append(myBridge.elements[damage_index[1]-1:damage_index[1]])

undamaged_elements = []  # Replace with your element labels
# Loop over all elements in the part
for i in range(len(myBridge.elements)):
    # If the type of the element is 'T3D2', append it to undamaged_elements
    if myBridge.elements[i].type == T3D2 and i != damage_index[0]-1 and i != damage_index[1]-1:
        undamaged_elements.append(myBridge.elements[i:i+1])

undamaged_truss_region = myBridge.Set(elements=undamaged_elements, name='Undamaged Truss Region')
myBridge.SectionAssignment(region = undamaged_truss_region, sectionName = 'Truss')

damaged_truss_region = myBridge.Set(elements=damaged_elements, name='Damaged Truss Region')
myBridge.SectionAssignment(region = damaged_truss_region, sectionName = 'Damaged Truss')

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DEFINING ANALYSIS JOB
import job

# Create an analysis job for the model and submit it.
jobName = 'bridge_analysis'
myJob = mdb.Job(name=jobName, model='Bridge', description='Bridge Analysis')

# Wait for the job to complete.
myJob.submit()
myJob.waitForCompletion()

from odbAccess import *
import matplotlib.pyplot as plt

# Open the output database
odb = openOdb(path=jobName + '.odb')

# Get the reference point set
refPointSet = odb.rootAssembly.nodeSets['LOADSET']

import numpy as np
# Get the displacement over time for the reference point
displacement = []
time = []
for i in range(0, 538):
    step = odb.steps['Harmonic Force']
    frames = step.frames
    time = [frame.frameValue for frame in frames]
    displacement_value = [frame.fieldOutputs['U'].getSubset(region=refPointSet).values[i].data for frame in frames] + odb.rootAssembly.instances['ALL-1'].nodes[i].coordinates
    displacement.append(displacement_value)

time = np.reshape(time, (1, 101, 1))
time = np.repeat(time, 538, axis=0)
data = np.dstack((displacement, time))

#print(np.shape(data))
np.save('D:\ABAQUS Jobs\Data_coordinates\displacement_data_2damageelement'+ str(damage_index[0]) + '-' + str(damage_index[1]) + '.npy', data)
#np.save('D:\ABAQUS Jobs\Data_coordinates\displacement_data_undamaged.npy', data)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import visualization
# Open the output database and display a
# default contour plot.
myViewport = session.Viewport(name='Bridge Analysis',
    origin=(20, 20), width=150, height=120)

myOdb = visualization.openOdb(path=jobName + '.odb')
myViewport.setValues(displayedObject=myOdb) 
myViewport.view.setValues(session.views['Iso'])
myViewport.odbDisplay.display.setValues(plotState=CONTOURS_ON_DEF)
myViewport.odbDisplay.commonOptions.setValues(renderStyle=FILLED)
myViewport.odbDisplay.setPrimaryVariable(variableLabel='S', outputPosition=INTEGRATION_POINT, refinement=(INVARIANT, 'Mises'), )
myViewport.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF, ))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#closeMdb(askSave=OFF)
#quit()
