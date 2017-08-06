# Cargar librerías import pandas as pd
import pandas as pd
import webhoseio
import string
import openpyxl

webhoseio.config(token="90a4a1a8-5016-4023-bdd6-8b302321a632")

#Cargar lista de empresas a buscar
datos = pd.read_csv('companylist.csv',delimiter=';')

empresas = datos['Nombre']
simbolos = datos['Symbol']

salida = pd.DataFrame()

for empresa, simbolo in zip(empresas, simbolos):
    empresa=empresa.replace(", Inc.","")
    empresa=empresa.replace(", Inc","")
    empresa=empresa.replace("Inc.","")
    empresa=empresa.replace("Inc","")
    query_params = {
    "q": "\"" + empresa + "\" site_type:news language:english (site:cnn.com OR site:wsj.com OR site:forbes.com OR site:marketwatch.com OR site:thestreet.com OR site:thisismoney.co.uk OR site:kiplinger.com site:bloomberg.com OR site:highpointobserver.com)",
    "ts": "1499376323267",
    "sort": "relevancy"
    }
    output = webhoseio.query("filterWebContent", query_params)
    output = pd.DataFrame(output['posts'])
    #output['text'] = output['text'].replace('\n','')
    output['symbol'] = simbolo
    if output.shape[0] > 0:
        for i in range(output.shape[0]):
            output.loc[i,'text'] = output.loc[i,'text'].replace('\n','')
            output.loc[i,'published'] = output.loc[i,'published'][0:10]
        salida = salida.append(output[['published','text','symbol']],ignore_index = True)

#salida.to_excel('dataset')
writer = pd.ExcelWriter('output.xlsx')
salida.to_excel(writer,'Sheet1')
writer.save()
