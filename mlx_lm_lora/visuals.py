class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_BLUE = '\033[44m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_RED = '\033[41m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

def print_banner():
    """Print a beautiful ASCII banner"""
    banner = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════════════════╗{Colors.RESET}
{Colors.CYAN}║{Colors.RESET}                                                                                          {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET} {Colors.BOLD}{Colors.MAGENTA}███╗   ███╗██╗     ██╗  ██╗    ██╗     ███╗   ███╗    ██╗      ██████╗ ██████╗  █████╗{Colors.RESET}   {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET} {Colors.BOLD}{Colors.MAGENTA}████╗ ████║██║     ╚██╗██╔╝    ██║     ████╗ ████║    ██║     ██╔═══██╗██╔══██╗██╔══██╗{Colors.RESET}  {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET} {Colors.BOLD}{Colors.BLUE}██╔████╔██║██║      ╚███╔╝     ██║     ██╔████╔██║    ██║     ██║   ██║██████╔╝███████║{Colors.RESET}  {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET} {Colors.BOLD}{Colors.BLUE}██║╚██╔╝██║██║      ██╔██╗     ██║     ██║╚██╔╝██║    ██║     ██║   ██║██╔══██╗██╔══██║{Colors.RESET}  {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET} {Colors.BOLD}{Colors.CYAN}██║ ╚═╝ ██║███████╗██╔╝ ██╗    ███████╗██║ ╚═╝ ██║    ███████╗╚██████╔╝██║  ██║██║  ██║{Colors.RESET}  {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET} {Colors.BOLD}{Colors.CYAN}╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝    ╚══════╝╚═╝     ╚═╝    ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝{Colors.RESET}  {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET}                                                                                          {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET} {Colors.YELLOW}{Colors.BOLD}Advanced Fine-Tuning Framework{Colors.RESET}                                                           {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET} {Colors.DIM}{Colors.WHITE}LoRA • (Online-)DPO • XPO • CPO • CPO • ORPO • PPO • GRPO • DrGRPO • GSPO • RLHF • SFT{Colors.RESET}   {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}║{Colors.RESET}                                                                                          {Colors.CYAN}║{Colors.RESET}
{Colors.CYAN}╚══════════════════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
    """
    print(banner)


def print_info(message):
    """Print info message in blue"""
    print(f"{Colors.BLUE}[INFO]{Colors.RESET} {message}")

def print_success(message):
    """Print success message in green"""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.RESET} {message}")

def print_warning(message):
    """Print warning message in yellow"""
    print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} {message}")

def print_error(message):
    """Print error message in red"""
    print(f"{Colors.RED}[ERROR]{Colors.RESET} {message}")

def print_section(title):
    """Print a section header"""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.WHITE}{title.center(60)}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")