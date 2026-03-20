content = open('generate_training_data.py', encoding='utf-8').read()
print(f"File loaded: {len(content.split(chr(10)))} lines")

# ── FIND the place_rooms function boundaries ───────────────────
lines = content.split('\n')
start_line = None
end_line = None
for i, line in enumerate(lines):
    if 'def place_rooms(' in line:
        start_line = i
    if start_line and i > start_line and line.startswith('def ') and 'place_rooms' not in line:
        end_line = i
        break

if end_line is None:
    end_line = len(lines)

print(f"place_rooms: lines {start_line+1} to {end_line}")
print(f"First line after: {lines[end_line] if end_line < len(lines) else 'EOF'}")

# Show what comes after place_rooms
print("\nLines after place_rooms function:")
for i in range(end_line, min(end_line+5, len(lines))):
    print(f"  {i+1}: {lines[i]}")
