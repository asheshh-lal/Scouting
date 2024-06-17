## Make sure you set a virtual env in your end and install requirements 

> python3 -m venv env_name

> pip install -r requirements.txt

### To run the app

> uvicorn main:app --reload

For API docs
> localhost:8000/docs


ProScout
ProScout is an innovative application designed to assist football teams in identifying suitable replacement players. It functions as a platform that leverages advanced algorithms to recommend 10 players who closely match the attributes of the players selected. Imagine a scenario where your talented left winger expresses a desire to leave your club during the upcoming transfer window due to wage constraints. While you wish for the player to stay, the club's financial limitations necessitate finding players with similar skills and characteristics. Traditionally, this process involved communicating the requirements to our talent acquisition team or scouts, who would then embark on a manual search based on historical data. This entailed physically attending matches, analyzing player performances, and returning with their findings. However, ProScout revolutionizes this process by swiftly recommending 10 comparable players. This not only expedites the player search but also generates a comprehensive report using Gemenai AI. This report offers a detailed comparison between the original player and the selected candidate, providing valuable insights for decision-making.

1. Our methodology for identifying similar players begins with meticulously gathering and scaling player data. We then employ cosine similarity algorithms to compute the similarity index, allowing us to pinpoint the top ten players who closely align with our criteria.

2. Furthermore, to provide a comprehensive analysis, we utilize Gemenai LLM (Language Learning Model) to generate detailed reports. These reports offer a nuanced understanding of the strengths, weaknesses, and overall playing styles of both the original player and the selected candidate. This comparative analysis serves as a valuable tool for clubs, enabling them to make informed decisions regarding player acquisitions and squad management.

![ezgif com-video-to-gif-converter](https://github.com/asheshh-lal/Scouting/assets/87692027/5f287d13-10dd-46bd-b051-2083c79f701d)
