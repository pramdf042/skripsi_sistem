mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"200411100106@student.trunojoyo.ac.id\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
