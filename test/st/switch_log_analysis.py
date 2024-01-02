valid_info_prefixes = ["model predict time is:", "model loaded, time:", "step:",
                    #    "input index:", "value:", "dtype:", "shape:"
                       ]

with open('switch_time.log') as switch_log_file:
    for line in switch_log_file.read().splitlines():
        is_valid_line = False
        for valid_prefix in valid_info_prefixes:
            if line.startswith(valid_prefix):
                is_valid_line = True
                break
        if not is_valid_line:
            continue
        print(line)