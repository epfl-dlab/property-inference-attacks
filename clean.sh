read -p "You are going to erase all saved logs and results. Do you want to proceed? (y/n):  " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]
then
  rm logs/*
  rm results/*
fi