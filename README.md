# Optical Character Recognition

- Se construiesc 10 arbori de decizie folosind functia build din RandomForest.
- Pentru fiecare dintre cei 10 arbori de decizie se alege random cate un sample,
  folosind functia get_random_samples din RandomForest.
- Fiecare arbore este antrenat in functia train din DecisionTree. Daca testele
  primite au aceeasi clasa, atunci se creeaza o frunza cu acea clasa ca
  rezultat.
- In caz contrar, cel mai bun split, mai exact cel care maximizeaza information
  gain, este determinat folosind functia find_best_split: fiecare dimensiune
  este considerata split index, iar media elementelor de pe coloana indexului de
  split este considerata split value, calculandu-se de fiecare data si entropia
  copiilor rezultati din split, iar in final se alege cel cu cel mai mare
  information gain.
- Se face split-ul, continuandu-se antrenarea copiilor.
- Daca nu s-a gasit un split valid, atunci se creeaza o noua frunza ce va avea
  ca rezultat clasa cu numar maxim de aparitii.
- In functia predict din RandomForest se interogheaza fiecare arbore de decizie
  si se retine rezultatul prezis, determinandu-se clasa cu cel mai mare numar de
  aparitii.

  Scheletul temei a fost implementat de studenti din echipa de SD
(Andrei Medar, Luca Istrate).
