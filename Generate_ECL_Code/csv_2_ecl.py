from os import listdir
import pandas as pd

result_folder = "./ECL_Classification/"
folder_name = "./ClassificationData/"
def convert2ecl(filename):
    content = ""
    line_break = "\n"
    tab = "    "
    df = pd.read_csv(filename)
    headers = ['id'] + df.columns.tolist()[:-1] + ['class']
    assert(len(headers) == len(df.columns) + 1), "something is wrong"
    content += "IMPORT * FROM ML;" + line_break
    dataset_name = filename.split("/")[-1].split("_")[-1].replace(".csv", "")
    content += "EXPORT " + dataset_name + "DS := MODULE " + line_break
    content += tab + "SHARED " + dataset_name + "RECORD := RECORD" + line_break
    for header in headers:
        content += tab*2 + "Types.t_FieldNumber " + header + ";" + line_break
    content += tab + "END;" + line_break
    content += tab + "EXPORT content := DATASET([" + line_break
    for lineno in range(len(df)):
        content += tab * 2 + "{" + str(lineno+1) + ", " + ",".join(map(str, df.iloc[lineno].tolist()))+ "}," + line_break

    # Remove the ","
    content = content[:-2] + line_break
    content += "]," +  dataset_name + "RECORD);" + line_break
    content += "END;" + line_break

    result_filename = result_folder + dataset_name + "DS.ecl"
    print ">> ", result_filename
    fd = open(result_filename, "w")
    fd.write(content)
    fd.close()


files = [folder_name + file for file in listdir(folder_name) if "csv" in file]

for i, file in enumerate(files):
    print i, file, "   ",
    convert2ecl(file)

