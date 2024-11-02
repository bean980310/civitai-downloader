def format_bytes(size):
    power=1024
    n=0
    power_labels=['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB']
    while size >= power and n < len(power_labels)-1:
        size/=power
        n+=1
    return f'{size:.2f} {power_labels[n]}'

def format_time(seconds):
    hours, remainder=divmod(seconds, 3600)
    minutes, seconds=divmod(remainder, 60)
    time_str=''
    if hours>0:
        time_str+=f'{int(hours)}h '
    if minutes>0:
        time_str+=f'{int(minutes)}m '
    time_str+=f'{int(seconds)}s'
    return time_str.strip()