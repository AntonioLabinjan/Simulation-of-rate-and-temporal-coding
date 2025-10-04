Simulation of rate and temporal coding

GPT SUMMARY:

### **1. Rate coding**

* Ideja: informacija se kodira **frekvencijom spikeova**.
* **Primjer:** imamo senzor (ili neuron) koji detektira intenzitet svjetla.

  * Kad je svjetlo jako → neuron šalje **više spikeova u jedinici vremena**.
  * Kad je svjetlo slabo → neuron šalje **manje spikeova**.
* Ključ: **vrijeme kada spike nastaje nije toliko važno**, važno je koliko ih je u određenom vremenskom periodu.
* Analogija: zamisli neku stanicu koja “raportira” intenzitet svijeta tako da šalje više ili manje pisama, ali ti te pisma možeš otvoriti bilo kada unutar tog perioda.

---

### **2. Temporal coding**

* Ideja: informacija se kodira **vremenom dolaska spikea** (tj. latencijom).
* **Primjer:** imamo isti senzor koji detektira intenzitet svjetla.

  * Jak signal → spike dolazi **odmah**.
  * Slab signal → spike dolazi **nakon odgode** (nekoliko ms kasnije).
* Ključ: **točno vrijeme spikea nosi informaciju**, a ne samo broj spikeova.
* Analogija: zamišljaj trku strijela. Što jači potez → strijela brže stigne do mete. Što slabiji → kasnije stigne, ali svejedno nosi informaciju o intenzitetu.

---

### **Razlika**

| Aspekt                           | Rate coding                                          | Temporal coding                                |
| -------------------------------- | ---------------------------------------------------- | ---------------------------------------------- |
| Što nosi informaciju             | Broj spikeova u vremenskom prozoru                   | Vrijeme dolaska spikea                         |
| Odnos s intenzitetom             | Veći intenzitet → više spikeova                      | Veći intenzitet → spike ranije                 |
| Osjetljivost na precizno vrijeme | Nije presudno                                        | Presudno                                       |
| Tipični slučajevi                | SNNs koji agregiraju po frekvenciji, manje real-time | Event-based senzori, precizno vrijeme je važno |

---

