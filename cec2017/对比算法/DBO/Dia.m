function [ y ] = Dia( arerfa,D )
%×Óº¯Êı ¶Ô½Ç¾ØÕó
y=zeros(D,D);
for i=1:D
    y(i,i)=arerfa^((i-1)/(2*(D-1)));
end


end

