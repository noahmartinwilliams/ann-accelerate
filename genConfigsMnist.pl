#! /usr/bin/swipl -q

:- initialization(main).

layer_sizes(15).
layer_sizes(16).
layer_sizes(32).
%layer_sizes(64).
%layer_sizes(128).

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

build_layer(Size, AF):- layer_sizes(Size), act_funcs(AF).

build_layers_intern(1, Layer5):-
	build_layer(Size, AF),
	Layer = '[',
	atom_concat(Layer, AF, Layer2),
	atom_concat(Layer2, ' ', Layer3),
	atom_concat(Layer3, Size, Layer4),
	atom_concat(Layer4, ']', Layer5).

build_layers_intern(X, Layer6):-
	X \= 1,
	build_layer(Size, AF),
	Layer = '[',
	atom_concat(Layer, AF, Layer2),
	atom_concat(Layer2, ' ', Layer3),
	atom_concat(Layer3, Size, Layer4),
	atom_concat(Layer4, '], ', Layer5),
	X2 is X - 1,
	build_layers_intern(X2, Rest),
	atom_concat(Layer5, Rest, Layer6).

build_layers(Layers3):- 
	num_layers(NumLayers),
	build_layers_intern(NumLayers, Layers0),
	atom_concat('[', Layers0, Layers2),
	atom_concat(Layers2, ']', Layers3).

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
