# DIG
NLP-Powered Data Insight Generator

The objective is to develop an app that can generate key metrics and visualizations for a variety of data analytics categories including "Sales Analytics", "Customer/User Analytics", "Financial Analytics", "Marketing Analytics", "Service Analytics", "Game Analytics", "Healthcare Analytics", "Logistics Analytics", "Social Media Analytics", "Risk Analytics".
Here is what the app does. First, it asks you to upload a csv file of your data. Then, it displays first 5 rows of data providing an overview of the data. Then, there is a drop-down menu that gives you an option to choose one of the ten categories of data analytics based on the type of your data.
The app searches for similarities in your columns’ name and default column names for the selected category to find a match. If there are matches for the columns involved in the key metrics and visualizations, key metrics and visualizations will be displayed. Then, a box appears that let you write your question about the data, and the app tries to answer it even if there are minor typos in your question.
Here are typical types of questions that app can handle:
•	Generating different types of plots: columns A versus column B
•	Statistical analysis of a columns such as unique values, min, max, total, average
•	Filtering based on date
•	Sorting, Grouping
•	Null detection
•	Number of rows and columns, column names

After answering a question, another box comes up and you can repeat asking questions until you type “no”, “no more”, “thank you”, or “done”.

Here is an example for questions regarding “game analytics”:
Q: column names
A: Column names: Date, Player_ID, Game_Session_ID, Game_Level, Session_Duration, In_Game_Purchases, Revenue, Retention_Rate, Player_Score, Highest_Score, Achievement_Unlocked, Date_year, Date_month, Date_day, Month_Year

Q: average game purchas
A: 24.95 (answer is generated based on the column name that is similar to question)

Q: pie plot max game purchas per gme level
A: Generated pie plot of In_Game_Purchases by Game_Level
