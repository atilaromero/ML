cat 20*/ex* | grep -v Name | awk '{print $1}' | sort -u | while read i
do
    grep -w $i 20*/ex* | awk '{print $6}' | xargs echo $i
done | sed -e 's/ /\t/' | sort -k 2
