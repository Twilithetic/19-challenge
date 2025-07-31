import sys
def get_single_char():
    """获取单个字符输入，无需按回车键"""
    try:
        # Windows系统
        import msvcrt
        return msvcrt.getch().decode('utf-8')
    except ImportError:
        # Unix/Linux/macOS系统
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

if __name__ == "__main__":
    print("请按任意键（无需按回车）：")
    char = get_single_char()
    print(f"你按下的字符是：{char} (ASCII码：{ord(char)})")