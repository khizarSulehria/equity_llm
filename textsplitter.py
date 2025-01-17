from langchain_text_splitters import CharacterTextSplitter

text = """
The stock of NVIDIA Corp (NASDAQ:NVDA) experienced a daily loss of -3.56% and a 3-month gain of 32.35%. With an Earnings Per Share (EPS) (EPS) of $1.92, the question arises: is the stock significantly overvalued? This article aims to provide a detailed valuation analysis of NVIDIA, offering insights into its financial strength, profitability, growth, and more. We invite you to delve into this comprehensive analysis.

Company Overview
Warning! GuruFocus has detected 10 Warning Signs with NVDA. Click here to check it out.

NVDA 30-Year Financial Data

The intrinsic value of NVDA


NVIDIA Corp (NASDAQ:NVDA) is a leading designer of discrete graphics processing units that enhance the experience on computing platforms. The firm's chips are widely used in various end markets, including PC gaming and data centers. In recent years, NVIDIA has broadened its focus from traditional PC graphics applications such as gaming to more complex and favorable opportunities, including artificial intelligence and autonomous driving, leveraging the high-performance capabilities of its products.

Currently, NVIDIA's stock price stands at $418.01, significantly higher than the GF Value of $310.28, indicating the stock might be overvalued. With a market cap of $1 trillion, the valuation seems steep. The following analysis aims to delve deeper into the company's value.

Is NVIDIA's Stock Significantly Overvalued? A Comprehensive Valuation Analysis
Is NVIDIA's Stock Significantly Overvalued? A Comprehensive Valuation Analysis
Understanding the GF Value
The GF Value is a unique measure of the intrinsic value of a stock, calculated based on historical trading multiples, a GuruFocus adjustment factor, and future business performance estimates. If the stock price is significantly above the GF Value Line, it is overvalued, and its future return is likely to be poor. Conversely, if it is significantly below the GF Value Line, its future return will likely be higher.

According to GuruFocus Value calculation, NVIDIA (NASDAQ:NVDA) appears to be significantly overvalued. The stock's current price of $418.01 per share and the market cap of $1 trillion further strengthen this assumption.

Given that NVIDIA is significantly overvalued, the long-term return of its stock is likely to be much lower than its future business growth.

Is NVIDIA's Stock Significantly Overvalued? A Comprehensive Valuation Analysis
Is NVIDIA's Stock Significantly Overvalued? A Comprehensive Valuation Analysis
Link: These companies may deliver higher future returns at reduced risk.

Financial Strength of NVIDIA
Examining the financial strength of a company is crucial before investing in its stock. Companies with poor financial strength pose a higher risk of permanent loss. NVIDIA's cash-to-debt ratio of 1.27 is worse than 58.04% of companies in the Semiconductors industry. However, NVIDIA's overall financial strength is 8 out of 10, indicating a strong financial position.

Is NVIDIA's Stock Significantly Overvalued? A Comprehensive Valuation Analysis
Is NVIDIA's Stock Significantly Overvalued? A Comprehensive Valuation Analysis
Profitability and Growth
Consistent profitability over the long term reduces the risk for investors. NVIDIA, with its profitability ranking of 10 out of 10, has been profitable for the past 10 years. The company's operating margin of 17.37% ranks better than 76.5% of companies in the Semiconductors industry.

However, growth is a crucial factor in a company's valuation. NVIDIA's growth ranks worse than 52.99% of companies in the Semiconductors industry, with its 3-year average revenue growth rate better than 87.88% of companies in the industry.

ROIC vs WACC
Comparing a company's return on invested capital (ROIC) to its weighted average cost of capital (WACC) is an effective way to evaluate its profitability. Over the past 12 months, NVIDIA's ROIC was 20.32 while its WACC was 16.74, suggesting that the company is creating value for its shareholders.

Is NVIDIA's Stock Significantly Overvalued? A Comprehensive Valuation Analysis
Is NVIDIA's Stock Significantly Overvalued? A Comprehensive Valuation Analysis
Conclusion
In conclusion, NVIDIA (NASDAQ:NVDA) appears to be significantly overvalued. Despite its strong financial condition and profitability, its growth ranks lower than 52.99% of companies in the Semiconductors industry. To learn more about NVIDIA stock, you can check out its 30-Year Financials here.

To find out the high quality companies that may deliver above-average returns, please check out GuruFocus High Quality Low Capex Screener.

This article first appeared on GuruFocus.
"""
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=10,
)
chunks = text_splitter.split_text(text)
# print(len(chunks))
#
# for chunk in chunks:
#     print(len(chunk))
#
# print(chunks[0])
# print("==============================")
# print(chunks[1])

from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ","
    ],
    chunk_size=1000,
    chunk_overlap=50
)

chunks = text_splitter.split_text(text)
print(len(chunks))

for chunk in chunks:
    print(len(chunk))

print(chunks[0])
print("==============================")
print(chunks[1])
print("==============================")
print(chunks[2])
