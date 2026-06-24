#! /usr/bin/swipl -q

:- initialization(main, main).

%layer_sizes(15).
layer_sizes(16).
layer_sizes(32).
layer_sizes(64).
%layer_sizes(128).
%layer_sizes(256).
%layer_sizes(512).
%layer_sizes(1024).

num_epochs(1).
num_epochs(8).

mini_batch_size(1).
mini_batch_size(2).
mini_batch_size(8).
mini_batch_size(16).

num_layers(1).
num_layers(2).
%num_layers(3).

act_funcs('Relu').
act_funcs('Sigmoid').
%act_funcs('Ident').
act_funcs('SoftMax').
%act_funcs('TanH').

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

bite_sizes(64).
bite_sizes(32).
%bite_sizes(16).
%bite_sizes(4).

concat_atoms([], '').
concat_atoms([A|B], Atom):- atom(A), concat_atoms(B, Rest), atom_concat(A, Rest, Atom).

build_layers(AFs):- num_layers(L), build_layers_intern(L, AFs2), concat_atoms(['[', AFs2, ']'], AFs).

build_layers_intern(0, '').
build_layers_intern(1, Atom):- !, build_layer(Atom).
build_layers_intern(X, L):- build_layer(Layer), X2 is X - 1, build_layers_intern(X2, L2), concat_atoms([Layer, ',', L2], L).

build_layer(AFs5):- layer_sizes(Size), build_layer_intern(Size, AFs), sort(AFs, AFs2), build_layer_post(AFs2, AFs3), build_layer_post2(AFs3, AFs4), concat_atoms(['[', AFs4], AFs5).

build_layer_post2([], '').
build_layer_post2([layer(AF, Size)], Atom):- !, atom_number(SizeAtom, Size), concat_atoms(['(', SizeAtom, ',', AF, ')', ']' ], Atom).
build_layer_post2([layer(AF, Size)|Rest], Atoms3):- atom_number(SizeAtom, Size), concat_atoms(['(', SizeAtom, ',', AF, '),'], Atoms), build_layer_post2(Rest, Atoms2), atom_concat(Atoms, Atoms2, Atoms3).

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

mk_ann(MBS, 'Adam', Layers, LR, Beta1, Beta2, E, InputAf):- mini_batch_size(MBS), num_epochs(E), build_layers(Layers), learn_rate(LR), beta1(Beta1), beta2(Beta2), act_funcs(InputAf).
mk_ann(MBS, 'SGD', Layers, LR, '0.9', '0.999', E, InputAf):- mini_batch_size(MBS), num_epochs(E), build_layers(Layers), learn_rate(LR), act_funcs(InputAf).
%mk_ann('RMSProp', Layers, LR, Beta1, ''):- build_layers(Layers), learn_rate(LR), beta1(Beta1).
%mk_ann('Adagrad', Layers, LR, '', ''):- build_layers(Layers), learn_rate(LR).
%mk_ann('Mom', Layers, LR, Beta1, ''):- build_layers(Layers), learn_rate(LR), beta1(Beta1).

ann2str(MBS, Epochs, Optim, Layers, LR, Beta1, Beta2, IAF, Tmp18) :- !,
	cost_fn(Cost),
	atom_concat('{"optimizer":"', Optim, Tmp1), 
	atom_concat(Tmp1, '","layers":"', Tmp2),
	build_layers(Layers),
	atom_concat(Tmp2, Layers, Tmp3),
	atom_concat(Tmp3, '","lr":', Tmp4),
	atom_concat(Tmp4, LR, Tmp5),
	atom_concat(Tmp5, ',"beta1":', Tmp6),
	atom_concat(Tmp6, Beta1, Tmp7),
	atom_concat(Tmp7, ',"beta2":', Tmp8),
	atom_concat(Tmp8, Beta2, Tmp9),
	atom_concat(Tmp9, ',"costF":"', Tmp10),
	atom_concat(Tmp10, Cost, Tmp11),
	atom_concat(Tmp11, '","numEpochs":', Tmp12),
	atom_concat(Tmp12, Epochs, Tmp13),
	atom_concat(Tmp13, ',"miniBatchSize":', Tmp14),
	atom_concat(Tmp14, MBS, Tmp15),
	atom_concat(Tmp15, ',"inputAF":"', Tmp16),
	atom_concat(Tmp16, IAF, Tmp17),
	atom_concat(Tmp17, '"}', Tmp18).

main:- mk_ann(MBS, Optim, Layers, LR, Beta1, Beta2, Epochs, IAF), 
	Prob is 1.0/1000.0, maybe(Prob), /*There are so many possible combinations that we (probably) can not fit them all on disk, so we need to cut out some of them at random.*/
	ann2str(MBS, Epochs, Optim, Layers, LR, Beta1, Beta2, IAF, Str), 
	catch(writeln(Str), _, halt(0)), 
	fail.
main:- halt(0).
