content = open('generate_training_data.py', encoding='utf-8').read()

old = '    cursor_y = net_d  # start from top, go down'
new = '    # Service zone starts below verandah, not at net_d\n    vd = dims.get("verandah", (net_w, 1.8))[1]\n    cursor_y = round(net_d - vd, 2)'

if old in content:
    content = content.replace(old, new)
    open('generate_training_data.py', 'w', encoding='utf-8').write(content)
    print('FIXED: service zone cursor now starts below verandah')
else:
    print('STRING NOT FOUND - showing context around cursor_y = net_d:')
    for i, line in enumerate(content.split('\n')):
        if 'cursor_y' in line and 'net_d' in line:
            print(f'  line {i}: {repr(line)}')
