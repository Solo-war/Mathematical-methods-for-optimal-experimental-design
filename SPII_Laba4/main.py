from owlready2 import *

# 1. Загрузка онтологии (убедитесь, что путь правильный и файл сохранен после исправлений!)
onto_path = r"SPII_Laba4/sports-club-ai.owl" # Замените на ваш реальный путь
onto = get_ontology(onto_path).load()

print("--- Классы онтологии ---")
for cls in onto.classes():
    print(cls.name)

print("\n--- Индивидуумы онтологии ---")
for ind in onto.individuals():
    print(ind.name)

# 2. Сведения о Petrov ДО reasoner
petrov = onto.Petrov
print(f"\n--- Сведения о {petrov.name} (ДО запуска reasoner) ---")
explicit_types = [cls.name for cls in petrov.is_a if hasattr(cls, 'name') and not str(cls).startswith('owl:')]
print("Явные типы:", explicit_types)

limitation_links = [lim.name for lim in petrov.hasLimitation]
print("Связи hasLimitation:", limitation_links)

# 3. Запуск логического вывода
print("\n* Owlready2 * Запуск HermiT...")
try:
    with onto:
        sync_reasoner() # Выполняет классификацию
    
    print(f"\n--- Сведения о {petrov.name} (ПОСЛЕ запуска reasoner) ---")
    inferred_types = [cls.name for cls in petrov.is_a if hasattr(cls, 'name') and not str(cls).startswith('owl:')]
    print("Все типы (включая выведенные):", inferred_types)
    
    # Проверим, появился ли нужный класс
    if "NeedsLoadAdjustment" in inferred_types:
        print("✅ УСПЕХ: Reasoner автоматически классифицировал Petrov как NeedsLoadAdjustment!")
    else:
        print("⚠️ ВНИМАНИЕ: Класс NeedsLoadAdjustment не был выведен. Проверьте факты у record_01.")
        
except OwlReadyJavaError as e:
    print("\n[ОШИБКА] Reasoner не смог запуститься.")
    print("Вы точно изменили xsd:date на xsd:string в Protégé и сохранили файл?")
    print("Текст ошибки:", e)

# 4. Проверка согласованности
print("\n--- Проверка согласованности ---")
inconsistent = list(onto.world.inconsistent_classes())
if inconsistent:
    print("❌ В онтологии обнаружены противоречия! Невыполнимые классы:", [c.name for c in inconsistent])
    print("Совет: Убедитесь, что вы удалили 'contains some Exercise' из класса Exercise.")
else:
    print("✅ Онтология непротиворечива (Consistent).")