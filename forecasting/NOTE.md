How to run the model
Run from project root

cd forecasting_project
python -m forecasting.cli --zone NL --train
python -m forecasting.cli --zone NL

# Push to github from Colab
!git init
!git add .
!git commit -m "Initial forecasting pipeline"
!git branch -M main
!git remote add origin https://github.com/yourname/your-repo.git
!git push -u origin main
