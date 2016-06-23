﻿EXPORT GenerateCode(experiment_name, algorithm, datasetNames, performance_scores, t_final_results):= MACRO
        #DECLARE(source_code)
        #SET(source_code, '');
        #DECLARE(indexs);
        #SET(indexs, 1);
        #LOOP
        	#IF(%indexs%> no_of_elements)	
        		#BREAK
        	#ELSE
                        #APPEND(source_code, 'result_' + datasetNames[%indexs%] + ' := ' + algorithm + '(' + QualifiedName(datasetNames[%indexs%]) + ', 2);\n');
                        #SET(indexs,%indexs%+1);
                #END
        #END
        #APPEND(source_code, 'final_results := DATASET([');
        #SET(indexs, 1);
        #LOOP
        	#IF(%indexs%>no_of_elements)	
        		#BREAK
        	#ELSE
                        #APPEND(source_code, '{' + %indexs% + ',\'' + datasetNames[%indexs%] + '\', result_' + datasetNames[%indexs%] + ',' + performance_scores[%indexs%] + '}');
                        #IF(%indexs%<no_of_elements)
                                #APPEND(source_code, ',\n');
                        #ELSE
                                #APPEND(source_code, '\n');
                        #END
                        #SET(indexs,%indexs%+1);
                #END
        #END
        #APPEND(source_code, '], dataset_record);\n');
        #APPEND(source_code, 'transormed_data_set_record := RECORD\n');
        #APPEND(source_code, 'final_results;\n');
        #APPEND(source_code, 'STRING Status;\n');
        #APPEND(source_code, 'END;\n');
        #APPEND(source_code, 'transormed_data_set_record assign_status(dataset_record L) := TRANSFORM\n');
        #APPEND(source_code, 'temp := ABS(L.ecl_performance - L.scikit_learn_performance);\n');
        #APPEND(source_code, 'SELF.Status := IF( temp <= 0.10, \'PASS\', \'FAIL\');\n');
        #APPEND(source_code, 'SELF:= L;\n');
        #APPEND(source_code, 'END;\n');
        #APPEND(source_code, 't_final_results := PROJECT(final_results, assign_status(LEFT));\n');
        //#APPEND(source_code, 'OUTPUT(t_final_results, NAMED(\''+ experiment_name + '\'));');
        #EXPAND(%'source_code'%);
ENDMACRO;