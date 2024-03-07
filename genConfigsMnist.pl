#! /usr/bin/swipl -q

:- initialization(main).

%layer_sizes(15).
layer_sizes(16).
layer_sizes(32).
layer_sizes(64).
%layer_sizes(128).
%layer_sizes(256).
%layer_sizes(512).
%layer_sizes(1024).

num_layers(1).
num_layers(2).
%num_layers(3).

act_funcs('Relu').
act_funcs('Sigmoid').
act_funcs('Ident').
act_funcs('Softmax').
act_funcs('TanH').

learn_rate('0.00001').
learn_rate('0.0001').
learn_rate('0.001').
learn_rate('0.01').

beta1('0.9').
beta1('0.99').
beta1('0.999').

beta2('0.9').
beta2('0.99').
beta2('0.999').

bite_sizes(32).
bite_sizes(16).
%bite_sizes(4).

concat_atoms([], '').
concat_atoms([A|B], Atom):- atom(A), concat_atoms(B, Rest), atom_concat(A, Rest, Atom).

build_layers(AFs):- num_layers(L), build_layers_intern(L, AFs2), concat_atoms(['[', AFs2, ']'], AFs).

build_layers_intern(0, '').
build_layers_intern(1, Atom):- !, build_layer(Atom).
build_layers_intern(X, L):- build_layer(Layer), X2 is X - 1, build_layers_intern(X2, L2), concat_atoms([Layer, ',', L2], L).

build_layer(AFs5):- layer_sizes(Size), build_layer_intern(Size, AFs), sort(AFs, AFs2), build_layer_post(AFs2, AFs3), build_layer_post2(AFs3, AFs4), concat_atoms(['[', AFs4], AFs5).

# transform entries into atoms.
build_layer_post2([], '').
build_layer_post2([layer(AF, Size)], Atom):- !, atom_number(SizeAtom, Size), concat_atoms([AF, ' ', SizeAtom, ']' ], Atom).
build_layer_post2([layer(AF, Size)|Rest], Atoms3):- atom_number(SizeAtom, Size), concat_atoms([' ' , AF, ' ', SizeAtom, ', '], Atoms), build_layer_post2(Rest, Atoms2), atom_concat(Atoms, Atoms2, Atoms3).

# find duplicates and remove them.
build_layer_post([], []):- !.
build_layer_post([layer(AF, Size1), layer(AF, Size2)|Rest], List):- !, Size3 is Size1 + Size2,  build_layer_post([layer(AF, Size3)|Rest], List).
build_layer_post([layer(AF, Size)|Rest], [layer(AF, Size)|List]):- build_layer_post(Rest, List).

build_layer_intern(Size, [layer(AF, BiteSize)|List]):- 
	bite_sizes(BiteSize),
	NextSize is Size - BiteSize,
	NextSize >= 0,
	act_funcs(AF),
	build_layer_intern(NextSize, List).
build_layer_intern(Size, [layer(AF, Size)]):-
	Size \= 0,
	act_funcs(AF).

build_layer_intern(0, []).




cost_fn('MSE').
cost_fn('CrossEntropy').

mk_ann('Adam', Layers, LR, Beta1, Beta2):- build_layers(Layers), learn_rate(LR), beta1(Beta1), beta2(Beta2).
mk_ann('SGD', Layers, LR, '', ''):- build_layers(Layers), learn_rate(LR).
mk_ann('RMSProp', Layers, LR, Beta1, ''):- build_layers(Layers), learn_rate(LR), beta1(Beta1).
mk_ann('Adagrad', Layers, LR, '', ''):- build_layers(Layers), learn_rate(LR).
mk_ann('Mom', Layers, LR, Beta1, ''):- build_layers(Layers), learn_rate(LR), beta1(Beta1).

ann2str(Optim, Layers, LR, Beta1, Beta2, Tmp14) :- !,
	cost_fn(Cost),
	act_funcs(AF),
	atom_concat('{ "optimizer":"', Optim, Tmp1), 
	atom_concat(Tmp1, '", "layers":"', Tmp2),
	atom_concat(Tmp2, Layers, Tmp3),
	atom_concat(Tmp3, '", "lr":"', Tmp4),
	atom_concat(Tmp4, LR, Tmp5),
	atom_concat(Tmp5, '", "beta1":"', Tmp6),
	atom_concat(Tmp6, Beta1, Tmp7),
	atom_concat(Tmp7, '", "beta2":"', Tmp8),
	atom_concat(Tmp8, Beta2, Tmp9),
	atom_concat(Tmp9, '", "costF":"', Tmp10),
	atom_concat(Tmp10, Cost, Tmp11),
	atom_concat(Tmp11, '", "inputAF":"', Tmp12),
	atom_concat(Tmp12, AF, Tmp13),
	atom_concat(Tmp13, '"}', Tmp14).

main:- mk_ann(Optim, Layers, LR, Beta1, Beta2), ann2str(Optim, Layers, LR, Beta1, Beta2, Str), writeln(Str), fail.
main:- halt(0).
