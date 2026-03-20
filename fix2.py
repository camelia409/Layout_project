content = open('generate_training_data.py', encoding='utf-8').read()

old = '    cursor_y = net_d'
new = '''    vd = dims.get("verandah", (net_w, 1.8))[1]
    cursor_y = round(net_d - vd, 2)'''

if old in content:
    # Only replace the first occurrence (in place_rooms, not elsewhere)
    content = content.replace(old, new, 1)
    open('generate_training_data.py', 'w', encoding='utf-8').write(content)
    print('FIXED')
else:
    print('NOT FOUND')
