;;; -*- Mode: Lisp; package:maxima; syntax:common-lisp ;Base: 10 -*- ;;;
;;; Translated on: 2020-07-02 20:20:44-03:00
;;; Maxima version: 5.43.2
;;; Lisp implementation: GNU Common Lisp (GCL)
;;; Lisp version: GCL 2.6.12
(in-package :maxima)
;;** Translator flags were: **

;; transcompile: true;
;; tr_semicompile: false;
;; translate_fast_arrays: false;
;; tr_warn_undeclared: compile;
;; tr_warn_meval: compfile;
;; tr_warn_fexpr: compfile;
;; tr_warn_mode: all;
;; tr_warn_undefined_variable: all;
;; tr_function_call_default: general;
;; tr_array_as_ref: true;
;; tr_numer: false;
;; define_variable: false;

(SIMPLIFY
    ($INTEGRATE
        (DIV 1.
             (ADD* (TRD-MSYMEVAL $S '$S)
                   (POWER (SIMPLIFY
                              (LIST '(%SQRT)
                                    (ADD*
                                     (POWER
                                      (ADD* (TRD-MSYMEVAL $V '$V)
                                       (*MMINUS (TRD-MSYMEVAL $X '$X)))
                                      2.)
                                     (POWER
                                      (ADD*
                                       (ADD*
                                        (MUL* (TRD-MSYMEVAL $M '$M)
                                         (TRD-MSYMEVAL $V '$V))
                                        (TRD-MSYMEVAL $B '$B))
                                       (*MMINUS (TRD-MSYMEVAL $Y '$Y)))
                                      2.))))
                          2.)))
        (TRD-MSYMEVAL $V '$V) (TRD-MSYMEVAL $L '$L)
        (TRD-MSYMEVAL $H '$H)))