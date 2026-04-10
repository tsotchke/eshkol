# Project: Build a Medical Diagnosis Expert System

A complete program that uses the consciousness engine to reason about
symptoms and suggest diagnoses. Combines knowledge bases, pattern
matching queries, and factor graph inference.

Save as `diagnosis.esk` and run with `eshkol-run diagnosis.esk -o diagnosis && ./diagnosis`.

---

## The Complete Program

```scheme
;; ═══════════════════════════════════════════════════════
;; Medical Diagnosis Expert System
;; Uses: KB (logic), factor graphs (inference)
;; ═══════════════════════════════════════════════════════

;; --- Build the knowledge base ---
(define kb (make-kb))

;; Symptom-disease associations
(kb-assert! kb (make-fact 'causes 'flu 'fever))
(kb-assert! kb (make-fact 'causes 'flu 'cough))
(kb-assert! kb (make-fact 'causes 'flu 'fatigue))
(kb-assert! kb (make-fact 'causes 'cold 'cough))
(kb-assert! kb (make-fact 'causes 'cold 'sneezing))
(kb-assert! kb (make-fact 'causes 'allergy 'sneezing))
(kb-assert! kb (make-fact 'causes 'allergy 'rash))
(kb-assert! kb (make-fact 'causes 'food-poisoning 'nausea))
(kb-assert! kb (make-fact 'causes 'food-poisoning 'fever))

;; Severity ratings
(kb-assert! kb (make-fact 'severity 'flu 'moderate))
(kb-assert! kb (make-fact 'severity 'cold 'mild))
(kb-assert! kb (make-fact 'severity 'allergy 'mild))
(kb-assert! kb (make-fact 'severity 'food-poisoning 'moderate))

;; Treatment recommendations
(kb-assert! kb (make-fact 'treatment 'flu 'rest-and-fluids))
(kb-assert! kb (make-fact 'treatment 'cold 'rest))
(kb-assert! kb (make-fact 'treatment 'allergy 'antihistamine))
(kb-assert! kb (make-fact 'treatment 'food-poisoning 'hydration))

;; --- Query engine ---

;; Find diseases that could cause a given symptom
(define (diseases-for-symptom symptom)
  (kb-query kb (make-fact 'causes ?disease symptom)))

;; Find all symptoms of a disease
(define (symptoms-of disease)
  (kb-query kb (make-fact 'causes disease ?symptom)))

;; Get treatment for a disease
(define (treatment-for disease)
  (kb-query kb (make-fact 'treatment disease ?treatment)))

;; --- Diagnosis ---

(display "=== Medical Diagnosis Expert System ===")
(newline) (newline)

;; Patient presents with fever and cough
(define patient-symptoms '(fever cough))

(display "Patient symptoms: ")
(display patient-symptoms)
(newline) (newline)

;; Find candidate diseases for each symptom
(display "Diseases that cause fever:")
(newline)
(display (diseases-for-symptom 'fever))
(newline)

(display "Diseases that cause cough:")
(newline)
(display (diseases-for-symptom 'cough))
(newline) (newline)

;; Check what flu causes — does it match our symptoms?
(display "All flu symptoms:")
(newline)
(display (symptoms-of 'flu))
(newline)

;; Get recommended treatment
(display "Treatment for flu:")
(newline)
(display (treatment-for 'flu))
(newline) (newline)

;; --- Probabilistic reasoning with factor graph ---
;; Model: symptom observations → disease probability

(define fg (make-factor-graph 3))

;; Factor: fever → flu (high correlation)
(fg-add-factor! fg 0 1 #(0.8 0.2 0.1 0.9))

;; Factor: flu → fatigue (secondary symptom)
(fg-add-factor! fg 1 2 #(0.7 0.3 0.3 0.7))

;; Run belief propagation
(fg-infer! fg 10)

(display "Factor graph inference complete.")
(newline)
(display "Free energy (surprise): ")
(display (free-energy fg #(0 0)))
(newline)

(display "=== Diagnosis: likely flu. Recommend rest and fluids. ===")
(newline)
```

---

## Key Concepts Demonstrated

1. **Knowledge base** — structured facts with predicate-argument patterns
2. **Logic variables** — `?disease`, `?symptom` in queries match any value
3. **Pattern matching queries** — find all facts matching a template
4. **Factor graphs** — probabilistic model of symptom-disease relationships
5. **Belief propagation** — `fg-infer!` propagates evidence through the graph
6. **Free energy** — quantifies how surprising the observations are given the model
