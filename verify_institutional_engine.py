
from strategy_generator import StrategyGenerator
from config import Config

def main():
    print("🚀 Verifying Institutional Engine Upgrade...")
    
    gen = StrategyGenerator()
    
    # Generate 10 strategies and look for institutional ones
    found_smc = False
    
    for i in range(20):
        s = gen._generate_single()
        print(f"Strat {i}: {s.name}")
        print(f"Rules: {s.entry_rules}")
        
        has_sweep = any("liquidity_sweep" in r for r in s.entry_rules)
        has_ob = any("order_block" in r for r in s.entry_rules)
        has_fvg = any("imbalance" in r for r in s.entry_rules)
        has_squeeze = any("volatility_squeeze" in r for r in s.entry_rules)
        
        if has_sweep or has_ob or has_fvg or has_squeeze:
            print("✅ FOUND INSTITUTIONAL STRATEGY!")
            found_smc = True
            break
            
    if found_smc:
        print("\nSUCCESS: The engine is generating complex Institutional matching patterns.")
    else:
        print("\nFAILURE: Only generated standard strategies (might be bad luck or weights issue).")

if __name__ == "__main__":
    main()
