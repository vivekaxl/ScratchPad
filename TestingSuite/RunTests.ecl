IMPORT Std;
IMPORT * FROM ML;
IMPORT ML.Tests.Explanatory as TE;
IMPORT * FROM ML.Types;
IMPORT * FROM TestingSuite.Utils;
IMPORT TestingSuite.Classification as Classification;

dataset_record := RECORD
	INTEGER dataset_id;
	STRING dataset_name;
	REAL ecl_performance;
        REAL scikit_learn_performance;
END;

QualifiedName(datasetname):= FUNCTIONMACRO
                RETURN 'Classification.Datasets.' + datasetname + '.content';
ENDMACRO;


SET OF STRING datasetNames := ['ecoliDS','GermanDS'];/*,'glassDS','houseVoteDS','ionosphereDS',
        'letterrecognitionDS','liverDS','ringnormDataDS','satimagesDS','segmentationDS', 
        'soybeanDS', 'VehicleDS', 'waveformDS'];*/   
SET OF REAL performance_scores := [0.818382927132, 0.748551052155, 0.703538248268, 0.963527328494, 0.926703202267,
                                        0.93904331765, 0.68403997834, 0.953616310418, 0.905047973842, 0.96728582798, 0.850413659417, 0.84263765448, 0.723008982605];
INTEGER no_of_elements := COUNT(datasetNames);
// sequential(
GenerateCode('RFClassifier', 'Classification.TestRandomForestClassification',  datasetNames, performance_scores, rf_results);
GenerateCode('DTClassifier', 'Classification.TestDecisionTreeClassifier',  datasetNames, performance_scores, dt_results);
OUTPUT(rf_results, NAMED('RandomForest'));
OUTPUT(dt_results, NAMED('DecisionTree'));
