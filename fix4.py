content = open('generate_training_data.py', encoding='utf-8').read()

old = '        mbw = min(mbw, prv_w_avail * 0.70)\n        mbd = min(mbd, prv_d_avail * 0.60)'
new = '        mbw = min(mbw, prv_w_avail * 0.90)\n        mbd = min(mbd, prv_d_avail * 0.45)'

if old in content:
    content = content.replace(old, new, 1)
    open('generate_training_data.py', 'w', encoding='utf-8').write(content)
    print('FIXED: master bedroom caps adjusted')
else:
    print('NOT FOUND')
    for i, line in enumerate(content.split('\n')):
        if 'prv_w_avail' in line and ('0.70' in line or '0.60' in line):
            print(f'  line {i+1}: {ascii(line)}')
