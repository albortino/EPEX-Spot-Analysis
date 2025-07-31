params_str = '\\"eww Wels V2\\", \\"!Restnetzbezug\\", \\"BeginDate\\", null, \\"yyyy-MM-dd HH:mm:ss\\",  (function (usage) {","    return parseFloat(usage.replace(\\",\\", \\".\\")'
# ---
params = []
current_param = ""
paren_depth = 0
bracket_depth = 0
in_string = False
string_char = None

i = 0
while i < len(params_str):
    char = params_str[i]
    
    if char == '\\': # Escape character
            i += 1
            continue
        
    if not in_string:
        if char in ['"', "'"]: # Within apostrophes
            in_string = True
            string_char = char
            i += 1
            continue
        elif char == '(': # Within parentheses
            paren_depth += 1
        elif char == ')':
            paren_depth -= 1
        elif char == '[': # Within brackets
            bracket_depth += 1
        elif char == ']':
            bracket_depth -= 1
        elif char == ',' and paren_depth == 0 and bracket_depth == 0:
            params.append(current_param.strip())
            current_param = ""
            i += 1
            continue
    else:
        if char == string_char: # and (i == 0 or params_str[i-1] != '\\')
            in_string = False
            string_char = None
            i += 1
            continue
    
    current_param += char
    i += 1

if current_param.strip():
    params.append(current_param.strip())
    
print(params, len(params))