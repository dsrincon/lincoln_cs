import Produccion_final_func as consejo
from tkinter import *

#Funciones
def kg_to_others():
    empresa = e1_value.get()
    advise = consejo.main(empresa)
    respuesta = 'The stock of ' + empresa + ' is most likely to go ' + advise + ' today.'
    t1.delete(1.0,END)
    t1.insert(END,respuesta)

#Crear vetnana
window = Tk()
#Label
l1 = Label(window, text="Company:")
l1.grid(row=0,column=0)
#Entry
e1_value = StringVar()
e1 = Entry(window, textvariable=e1_value)
e1.grid(row=0,column=1)
#Button
b1 = Button(window,text='Try me',command=kg_to_others)
b1.grid(row=0,column=2)
#Textos
t1 = Text(window,height=1,width=90) #Pounds
t1.grid(row=1,column=1)

window.mainloop()
