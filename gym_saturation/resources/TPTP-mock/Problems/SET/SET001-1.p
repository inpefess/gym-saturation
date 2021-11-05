%--------------------------------------------------------------------------
% This file is a part of TPTP. For more information see http://www.tptp.org/

% Copyright
% ---------
% The TPTP is copyrighted 1993-onwards,  by Geoff Sutcliffe &  Christian Suttner.
% Verbatim redistribution of the TPTP and parts of the TPTP is permitted provided
% that the redistribution is clearly attributed to the TPTP.  Distribution of any
% modified version or modified part of the TPTP requires permission.

% File     : SET001-1 : TPTP v7.5.0. Released v1.0.0.
% Domain   : Set Theory
% Problem  : Set members are superset members
% Version  : [LS74] axioms.
% English  : A member of a set is also a member of that set's supersets.

% Refs     : [LS74]  Lawrence & Starkey (1974), Experimental Tests of Resol
%          : [WM76]  Wilson & Minker (1976), Resolution, Refinements, and S
% Source   : [SPRFN]
% Names    : ls100 [LS74]
%          : ls100 [WM76]

% Status   : Unsatisfiable
% Rating   : 0.00 v2.0.0
% Syntax   : Number of clauses     :    9 (   1 non-Horn;   3 unit;   8 RR)
%            Number of atoms       :   17 (   0 equality)
%            Maximal clause size   :    3 (   2 average)
%            Number of predicates  :    3 (   0 propositional; 2-2 arity)
%            Number of functors    :    4 (   3 constant; 0-2 arity)
%            Number of variables   :   13 (   0 singleton)
%            Maximal term depth    :    2 (   1 average)
% SPC      : CNF_UNS_RFO_NEQ_NHN

% Comments :
%--------------------------------------------------------------------------
%----Include the member and subset axioms
include('Axioms/SET001-0.ax').
%--------------------------------------------------------------------------
cnf(b_equals_bb,hypothesis,
    ( equal_sets(b,bb) )).

cnf(element_of_b,hypothesis,
    ( member(element_of_b,b) )).

cnf(prove_element_of_bb,negated_conjecture,
    ( ~ member(element_of_b,bb) )).

%--------------------------------------------------------------------------
