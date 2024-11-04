
class DownloadUtils:
    @staticmethod
    def format_bytes(size):
        for unit in ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB']:
            if size<1024 or unit == 'PB':
                return f"{size:.2f} {unit}"
            size/=1024

    @staticmethod
    def format_time(seconds):
        h, rem=divmod(int(seconds), 3600)
        m, s=divmod(rem, 60)
        result=''
        if h>0:
            result+=f'{int(h)}h '
        if m>0:
            result+=f'{int(m)}m '
        result+=f'{int(s)}s'
        return result.strip()