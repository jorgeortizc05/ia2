import re
import itertools

class Utilities:
    
    def __init__(self, path = '/home/jorge/Documentos/ia2/ProyectoInterciclo/Digitos/corpus/digits-database.data'):
        self.path = path
        self.regex = re.compile('(0|1){2,}') # Busca patrones que coincidan con 2 o mas ceros o unos.
        self.regexno = re.compile('(\s)+[0-9]{1}') # Busca un unico numero el cual tenga un espacio o tabulacion antes del mismo.
        
    
    def generate_indices(self):
        _dict = []
        with open(self.path, 'r') as _f:
            pivote = 0
            flag = False
            lineno = 0
            for line in _f:
                if self.regex.match(line)!=None and not flag:
                    pivote = lineno
                    flag = True
                if self.regexno.match(line)!=None and flag:
                    _dict.append((int(line.replace(' ','')),pivote,lineno))
                    flag = False
                lineno += 1
            _f.close()
            
        return _dict

    def get_digit(self,_slice, _end):
        data = []
        with open(self.path, 'r') as _f:
            for line in itertools.islice(_f, _slice, _end):
                data.append([int(i) for i in line.lstrip().rstrip()])
            
            _f.close()
        return data
