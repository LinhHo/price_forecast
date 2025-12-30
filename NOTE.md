How to run the model
Run from project root

cd forecasting_project
python -m forecasting.cli --zone NL --train
python -m forecasting.cli --zone NL

# Push to github from Colab
git init
git add .
git commit -m "Initial forecasting pipeline"
git remote add origin <REMOTE_URL>

https://github.com/LinhHo/price_forecast.git

git branch -M main
git remote -v # verify that the remote URL was set correctly
git push -u origin main # Push your local commits to GitHub. This command pushes the changes in your current branch (commonly main or master) to the origin remote, and the -u flag sets the upstream branch for future pushes:


!git remote add origin https://github.com/yourname/your-repo.git
!git push -u origin main
