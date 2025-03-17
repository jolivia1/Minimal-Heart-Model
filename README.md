---- A MINIMAL ELECTRICAL MODEL OF THE HUMAN HEART ----

In this repository you find all of the code developed during the project.



All of the Python scripts are in the folder "/models". 



We found it easier to have separate scripts for the various case studies conducted:


--------------------------------------
-- Heart_model.py -- 

This is the main code which models the "healthy heart scenario".
This script has the standard structure from which the others were created.

The parameter I_value (input current at the SAN) may be tweaked to access different heart rates as described in the paper.


-- Ventricular_tachycardia.py -- 

Introduces a spiral wave in the left ventricle.

To achieve this structure the epicardial layer is removed.


-- AVNRT.py -- 

The AVN has an internal structure that connects to the bundle. This structure is shown alongside the whole heart view.

An artificial unidirectional block is introduced to mimic a premature excitation. 


-- WPW_syndrome.py -- 

An extra connection between the atria and the ventricles is created.

-- Ischemia.py -- 

A square region of random deficient diffusion is created in the right atrium. 

Given this randomness, it is impossible to tell which initial configurations will yield reentry.


--------------------------------------

all of these scipts show the geometry of the heart when run. Upon closing this view the iteration stops and a plot of the ECG of the 3 leads is shown.

Once the plot is closed a .png and a .csv file are saved, both relating to the ecg data.

The CSV has 4 columns:
1st column: time
2nd column: Lead I ECG voltage
3rd column: Lead II ECG voltage
4th column: Lead III ECG voltage


Thank you for your interest,
Joao Olivia and Rui Dilao
