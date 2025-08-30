from browser_use_sdk import BrowserUseSdk

sdk = BrowserUseSdk(api_key="your api key here")

task = """
You need to perform the following browser navigation related to the academic topic 'Differential Equations':

1. Open https://www.britannica.com in the browser.
2. Search for "Differential Equation" in the search bar.
3. Click on the article about "Differential Equations".
4. From there, follow the hyperlink to "Ordinary Differential Equation".
5. Then, navigate to the page on "Initial Conditions".
6. Next, click on the section discussing "Boundary Conditions".
7. Finally, move to the page about "Euler’s Numerical Method".

Make sure each step is completed in sequence and include a short note on what you found at each stage.
"""

result = sdk.run(
    llm_model="o3",
    task=task
)

print(result)
