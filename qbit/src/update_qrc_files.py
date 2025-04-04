#!/usr/bin/env python

import os
from os.path import splitext, join


# atualiza o diretório dos arquivos de idiomas
languages_list = [x for x in os.listdir('lang') if x.endswith('.qm')]
output = '''<!DOCTYPE RCC><RCC version="1.0">
<qresource>
'''

for language in languages_list:
    output += '  <file>%s</file>' % ('lang' + os.sep + language)
    output += os.linesep

output += '''</qresource>
</RCC>'''

lang_file = open('lang.qrc', 'w')
lang_file.write(output)
lang_file.close()


# atualiza o diretório de arquivos de ícones
icons_list = []

for root, dirs, files in os.walk('Icons'):
    if 'skin_unused' in dirs:
        dirs.remove('skin_unused')

    for file in files:
        if splitext(file)[-1] in ('.png', '.jpg'):
            icons_list.append(join(root, file))

output = '''<!DOCTYPE RCC><RCC version="1.0">
<qresource>
'''

for icon in icons_list:
    output += '  <file>%s</file>' % (icon)
    output += os.linesep

output += '''</qresource>
</RCC>'''

icons_file = open('icons.qrc', 'w')
icons_file.write(output)
icons_file.close()