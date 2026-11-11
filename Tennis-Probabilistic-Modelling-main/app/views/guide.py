import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st  # noqa: E402

from common import runtime, setup  # noqa: E402

setup("How it works")
m = runtime().metrics
b = m.get("market_benchmark", {})
new = ((m.get("backtests", {}).get("rules", {}).get("new_ev3")) or {})
best, half = new.get("best", {}), new.get("halfway", {})

st.markdown('<div class="hero"><h1>📖 How it works</h1><div class="sub">Five minutes, no maths degree needed.</div></div>',
            unsafe_allow_html=True)

st.markdown("### The idea in three sentences")
st.markdown(
    "1. **The sharpest bookmaker's price is the best public estimate of who wins.** Pinnacle takes the biggest bets from the best "
    "bettors, so its closing price is very hard to beat."
    + (f" In testing, its favourite won {b['pinnacle_accuracy']:.0%} of the time; this app's model alone got {b['model_accuracy']:.0%}."
       if b else "") + "\n"
    "2. **Other bookmakers are sometimes slow or generous.** When one of them pays clearly more than the sharp price says is fair, "
    "that bet is worth more than it costs.\n"
    "3. **So the app compares every book's price with the fair price and tells you when the gap is big enough**, how much to "
    "stake, and the lowest price worth taking. The model adds a second opinion and a small adjustment.")

st.markdown("### Your routine")
st.markdown(
    '<div class="steps">'
    '<div class="step"><div class="n">Once</div><div class="t">Bankroll and books</div><div class="d">Set aside money you can afford to '
    'lose and enter it as your bankroll. Tick only the sportsbooks you can actually bet with.</div></div>'
    '<div class="step"><div class="n">Each day</div><div class="t">Open “Today\'s bets”</div><div class="d">Green cards are bets. Phone '
    'alerts (optional, see the README) bring them to you the moment they appear.</div></div>'
    '<div class="step"><div class="n">Before betting</div><div class="t">Confirm the price</div><div class="d">Open your sportsbook. If '
    'the price is at or above “don\'t take below”, place the stake shown. If not, skip.</div></div>'
    '<div class="step"><div class="n">Weekly</div><div class="t">Check Results</div><div class="d">Watch “beat the closing price by”. '
    'Positive over a few hundred bets means it is working; negative means stop and rethink.</div></div>'
    '</div>', unsafe_allow_html=True)

st.markdown("### What the words mean")
g = st.columns(2)
g[0].markdown(
    "**Fair chance / fair price.** The app's best estimate of a player's chance to win, and the odds that would exactly break "
    "even at that chance. A 50% chance has a fair price of 2.00.\n\n"
    "**Value (EV).** Expected profit per dollar staked. A 50% chance at odds of 2.10 has a value of +5%: over many such bets you "
    "would expect to make about 5 cents per dollar.\n\n"
    "**Margin (vig).** Bookmakers shade their prices so both sides add up to more than 100%. The app removes this before "
    "comparing prices.")
g[1].markdown(
    "**Stake (Kelly).** A formula that bets more when the value is bigger and the price shorter. The app uses a quarter of it by "
    "default and never more than 2% of your bankroll on one bet.\n\n"
    "**Don't take below.** The lowest price at which the bet still passes the rule. Prices move; this is your cut-off.\n\n"
    "**Closing-line value (CLV).** Whether the price you took was better than the final price just before the match. It is the "
    "quickest honest test of whether your bets are good, long before profit settles down.")

st.markdown("### What to expect, honestly")
if best.get("bets"):
    st.markdown(
        f"Replayed on {m['backtests']['matches']:,} past matches ({m['backtests']['years'][0]}–{m['backtests']['years'][1]}), the default "
        f"rule returned **{best['flat_roi']:+.1%} per dollar staked (± {best['flat_roi_se']:.1%})** if you always got the best closing "
        f"price of any bookmaker, on {best['bets']:,} bets"
        + (f", and **{half['flat_roi']:+.1%} (± {half['flat_roi_se']:.1%})** on {half['bets']:,} bets at a price halfway between the "
           "average and the best" if half.get("bets") else "") + ". The Model page has the full table.")
st.markdown(
    "- **That is a ceiling, not a promise.** It assumes accounts at many books and that the top price is still there when you "
    "bet. With three or four books you will see fewer bets and a lower return.\n"
    "- **Losing runs are normal.** One season in the test lost money. Dozens of losing bets in a row will happen.\n"
    "- **Books limit winners.** Bookmakers other than Pinnacle and the exchanges cut the stakes of customers who keep taking "
    "their best prices.\n"
    "- **The edge is shrinking.** Qualifying bets per season have fallen as bookmakers' prices have converged.\n"
    "- **Grand Slam total sets** can't be back-tested (no historical prices exist), so treat those bets as experimental.\n"
    "- **Nothing here is a guarantee.** Only bet money you can afford to lose. If betting stops being fun or you are chasing "
    "losses, stop; in Ontario, ConnexOntario (1-866-531-2600) offers free, confidential help.")
