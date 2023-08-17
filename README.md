# Serwis reprezentatywności
## 1. Zadanie
W ramach zadania należy zaimplementować serwis REST’owy w języku Python, który
pozwoli na wyuczanie i następnie odpytywanie modelu o tak zwaną reprezentatywność
obiektu. Serwis został zaimplementowany przy użyciu technologi FastAPI. Wykorzystano następujące bibglioteki - numpy, 
sklearn, scipy oraz uvicorn.

## 2. Funkcjonalności
* Zlecenie wytrenowania nowego modelu na przesłanych danych.
* Sprawdzenie statusu ostatnio zleconego treningu.
* Predykcja ostatnio wytrenowanego modelu na przesłanych obiektach.

## 3. Opis mikroserwisu
#### API:
* _GET_ _/status_ -> zwraca status treningu
* _POST_ _/train_ -> zlecenie treningu modelu na przesłanych danych 
* _POST_ _/predict_ -> zlecenie predykcji na wcześniej wytrenowanym modelu

## 4. Uruchomienie
Uruchomienie serwisu bezpośrednio
```bash
uvicorn main:app 
```
lub poprzez dockera
```bash
docker build -t representation-service .
```
```bash
docker-compose up -d
```


